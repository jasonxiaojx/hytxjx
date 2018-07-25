'''
sample submission
参赛者提交代码示例
入参出参已公布
'''
import pandas as pd
import numpy as np
from scipy import stats    # 注意这里是新加的，上传时需要。
import functools 
import multiprocess as mp
import time

class UserPolicy:
    def __init__(self, initial_inventory, inventory_replenishment, sku_demand_distribution, sku_cost  ):
        self.inv = [initial_inventory]
        self.replenish = inventory_replenishment
        self.distribution = sku_demand_distribution
        self.cost = sku_cost
        self.sku_limit = np.asarray([200, 200, 200, 200, 200])
        self.extra_shipping_cost_per_unit = 0.01
        self.capacity_limit = np.asarray([3200, 1600, 1200, 3600, 1600])
        self.abandon_rate =np.asarray([1./100, 7./100, 10./100, 9./100, 8./100])

    def allocation_sku(fdc_mu, fdc_sigma, fdc_stock, rdc_available):
        lengh = fdc_mu.shape[0]
        if_valid = np.array([True] * lengh)
        #if_valid[1] = False
        tmp_amount = np.array([0] * lengh)
        if rdc_available < 1:
            return tmp_amount
        while True:
            z_star = (rdc_available + np.sum(fdc_stock[if_valid]) - \
                          np.sum(fdc_mu[if_valid])) / np.sum(fdc_sigma[if_valid])
            tmp_amount[if_valid] = fdc_mu[if_valid] + z_star * fdc_sigma[if_valid] - fdc_stock[if_valid]
            if_valid = (tmp_amount>=0) & if_valid
            # 判断是否存在小于0的
            if (tmp_amount>=0).sum() ==lengh:
                break
            else:
                tmp_amount[~if_valid] = 0
        # 检查四舍五入取整是否出现问题
        while np.round(tmp_amount).sum() > rdc_available:
            tmp_amount[if_valid] = tmp_amount[if_valid] - 0.1
        return np.round(tmp_amount).astype(int)
        
    def daily_decision(self,t):
        '''
        daily decision of inventory allocation
        input values:
            t: decision date
        return values:
            replenishment_decision, 2-D numpy array, shape (5,1000)
        '''

        # Your algorithms 

        limit = pd.DataFrame({ 'capacity_limit' : [3200, 1600, 1200, 3600, 1600], 'sku_limit' : [200, 200, 200, 200, 200],})
        abandon_rate = np.asarray([1./100, 7./100, 10./100, 9./100, 8./100])
        sku_demand_dist = self.distribution
        inventory = self.inv[-1]
        sku_cost = self.cost
        replenish = self.replenish
        replenish = replenish.sort_values(by=['item_sku_id','date'])

        
        def allocation_rdc(inventory,replenish,t):

            timestart = time.time()
            
            dm_invntry = pd.merge(sku_demand_dist, inventory, how='inner', on=('item_sku_id','dc_id'))
            # 计算下次最早采购达到时间
            replenish['flag'] = replenish.date.apply(lambda x: 1 if x>=t else 0)
            next_repln = pd.DataFrame(replenish[replenish['flag']==1]['date'].groupby(by=replenish.item_sku_id).min()).reset_index()
            dm_invntry = pd.merge(dm_invntry, next_repln, how='left', on=('item_sku_id'))
            dm_invntry['days_btw'] = dm_invntry['date'] - t + 1
            dm_invntry['days_btw'].fillna(31-t, inplace=True)
            
            # 销量分布的重要参数计算：mean, std，用flag来表示属于哪种分布。
            dm_invntry['flag_gm'] = dm_invntry.dist_type.apply(lambda x: 1 if x=='G' else 0) 
            dm_invntry['flag_nb'] = 1 - dm_invntry['flag_gm']
            dm_invntry['para1_adj'] = dm_invntry['para1'] * dm_invntry['days_btw']  # 生成采购到货天数内总销量的分布参数
            dm_invntry['gamma_m'] = stats.gamma.stats(a=dm_invntry['para1_adj'], scale=dm_invntry['para2'], moments='mvsk')[0]
            dm_invntry['gamma_sd'] = stats.gamma.stats(a=dm_invntry['para1_adj'], scale=dm_invntry['para2'], moments='mvsk')[1]**(1/2)
            dm_invntry['nbinom_m'] = stats.nbinom.stats(dm_invntry['para1_adj'], dm_invntry['para2'], moments='mvsk')[0]
            dm_invntry['nbinom_m'].fillna(0, inplace=True)
            dm_invntry['nbinom_sd'] = stats.nbinom.stats(dm_invntry['para1_adj'], dm_invntry['para2'], moments='mvsk')[1]**(1/2)
            dm_invntry['nbinom_sd'].fillna(0, inplace=True)
            dm_invntry['sd'] = dm_invntry['gamma_sd'] * dm_invntry['flag_gm'] + dm_invntry['nbinom_sd'] * dm_invntry['flag_nb']
            dm_invntry['exp_sales'] = dm_invntry['gamma_m'] * dm_invntry['flag_gm'] + dm_invntry['nbinom_m'] * dm_invntry['flag_nb']

            # 建立初始虚拟库存列，FDC的为真实的，RDC的为0
            dm_invntry['virtual_inv'] = dm_invntry['stock_quantity'].copy()
            dm_invntry['virtual_inv'][dm_invntry['dc_id']==0] = 0
            
            # RDC可分配库存量
            # 改为向上取整。
            Q = dict(dm_invntry[dm_invntry['dc_id'] == 0]['stock_quantity'].groupby(by=dm_invntry.item_sku_id, sort=False).sum() * 0.9)

            ################### 此处开始写新的分配策略 ####################
            # dm_invntry['Q'] = dm_invntry.item_sku_id.apply(lambda x: np.ceil(Q[x]*0.85))
            dm_invntry = dm_invntry.sort_values(['item_sku_id','dc_id']).set_index('item_sku_id')

            def allocation_sku(fdc_mu, fdc_sigma, fdc_stock, rdc_available):
                length = fdc_mu.shape[0]
                if_valid = np.array([True] * length)
                #if_valid[1] = False
                tmp_amount = np.array([0] * length)
                if rdc_available<1:
                    return tmp_amount
                while True:
                    z_star = (rdc_available + np.sum(fdc_stock[if_valid]) - \
                                  np.sum(fdc_mu[if_valid])) / np.sum(fdc_sigma[if_valid])
                    tmp_amount[if_valid] = fdc_mu[if_valid] + z_star * fdc_sigma[if_valid] - fdc_stock[if_valid]
                    if_valid = (tmp_amount>=0) & if_valid
                    # 判断是否存在小于0的
                    if (tmp_amount>=0).sum() ==length:
                        break
                    else:
                        tmp_amount[~if_valid] = 0
                # 检查四舍五入取整是否出现问题
                while np.round(tmp_amount).sum() > rdc_available:
                    tmp_amount[if_valid] = tmp_amount[if_valid] -0.1
                return np.round(tmp_amount).astype(int)
            
            def sku_iter(dm_invntry,sku):
                stock = dm_invntry.loc[sku]['virtual_inv'].values
                mu = dm_invntry.loc[sku]['exp_sales'].values
                sd = dm_invntry.loc[sku]['sd'].values
                return_list = []
                return_list.append([sku]*6)
                return_list.append(dm_invntry.loc[sku]['dc_id'].values.tolist())
                return_list.append(allocation_sku(mu, sd, stock, Q[sku]).tolist())

                return return_list
            
            partial_param = functools.partial(sku_iter,dm_invntry)
            pool = mp.Pool(processes=2)   # 4进程55s，5线程48s，
            #pool = Pool(processes=5)
            try:
                fdc_output = pool.map(partial_param, range(1,1001))
                pool.close()
                #pool.join()
            except KeyboardInterrupt as e:
                pool.terminate()
                raise e

            allocation_rdc = {'item_sku_id':fdc_output[0][0], 'dc_id':fdc_output[0][1], \
                                  'capacity_interger':fdc_output[0][2]}
            for j in range(1,1000):
                allocation_rdc['item_sku_id'].extend(fdc_output[j][0])
                allocation_rdc['dc_id'].extend(fdc_output[j][1])
                allocation_rdc['capacity_interger'].extend(fdc_output[j][2])
                
            print("single sku multi dc problem:",time.time() - timestart)
            return pd.DataFrame(allocation_rdc)

        
        def allocation_fdc_i(init_cr,limit,inventory,allocation_rdc_r,i):
            m_cost_sku_cr = init_cr  # 0.01
            m_cost_sku_cr_cp = init_cr
            step_cr = 0.1
            step_cr_cp = 0.1
            cp_lm = limit['capacity_limit'][i-1]
            sku_lm = limit['sku_limit'][i-1]
            sku_dm_fdc = sku_demand_dist[sku_demand_dist['dc_id']==i].copy()
            invntry_fdc = inventory[inventory['dc_id']==i].copy()
            dm_inv_fdc_t = pd.merge(sku_dm_fdc, invntry_fdc, how='inner', on=('item_sku_id','dc_id'))

            dm_inv_fdc_t['flag_gm'] = dm_inv_fdc_t.dist_type.apply(lambda x: 1 if x=='G' else 0)
            dm_inv_fdc_t['flag_nb'] = 1 - dm_inv_fdc_t['flag_gm']
            
            dm_inv_fdc_t['gamma_m'] = stats.gamma.stats(a=dm_inv_fdc_t['para1'], scale=dm_inv_fdc_t['para2'], moments='mvsk')[0]
            dm_inv_fdc_t['nbinom_m'] = stats.nbinom.stats(dm_inv_fdc_t['para1'], dm_inv_fdc_t['para2'], moments='mvsk')[0]
            dm_inv_fdc_t['nbinom_m'].fillna(0, inplace=True)
            dm_inv_fdc_t['exp_sales'] = dm_inv_fdc_t['gamma_m'] * dm_inv_fdc_t['flag_gm'] + \
                                    dm_inv_fdc_t['nbinom_m'] * dm_inv_fdc_t['flag_nb']
                
            # 设置一个20天目标库存，控制调拨节奏，不要全部都调过来了
            max_ti_days = 20
            dm_inv_fdc_t['max_ti'] = dm_inv_fdc_t['exp_sales'] * max_ti_days
            dm_inv_fdc_t['max_transfer'] = (dm_inv_fdc_t['max_ti'] - dm_inv_fdc_t['stock_quantity'])
            dm_inv_fdc_t['max_transfer'] = dm_inv_fdc_t['max_transfer'].apply(lambda x: x if x > 0 else 0)
            
            dm_inv_fdc_t = pd.merge(dm_inv_fdc_t, allocation_rdc_r, how='inner', on=('item_sku_id','dc_id'))
            dm_inv_fdc_t = pd.merge(dm_inv_fdc_t, sku_cost, how='inner', on=('item_sku_id'))
            
            dm_inv_fdc_t['cost_adj'] = dm_inv_fdc_t['stockout_cost']*abandon_rate[i-1]+0.01*(1-abandon_rate[i-1])
            dm_inv_fdc_t['m_cost'] = dm_inv_fdc_t.cost_adj.max()

            sku_cnt_fdc_t = 0
            capacity_cnt_fdc_t = 0
            k = 0
            while step_cr > 0.001:
                step_cr = step_cr/2
                while True:
                    k += 1
                    #print('FDC',i,'m_cost_sku_cr',m_cost_sku_cr,'step_cr',step_cr,'sku_cnt_fdc_t',sku_cnt_fdc_t,'sku_lm',sku_lm,'capacity_cnt_fdc_t',capacity_cnt_fdc_t,'cp_lm',cp_lm)
                    m_cost_sku_cr += step_cr
                    dm_inv_fdc_t['m_cost_sku_cr'] = m_cost_sku_cr
                    dm_inv_fdc_t['cr'] = 1-dm_inv_fdc_t['m_cost']/dm_inv_fdc_t['cost_adj']*(1-dm_inv_fdc_t['m_cost_sku_cr'])
                    #dm_inv_fdc_t['cr'] = 1-dm_inv_fdc_t['m_cost']/dm_inv_fdc_t['stockout_cost']*(1-dm_inv_fdc_t['m_cost_sku_cr'])

                    dm_inv_fdc_t['cr'] = dm_inv_fdc_t['cr'].apply(lambda x: 1 if x > 1 else (0 if x < 0 else x))

                    dm_inv_fdc_t['gamma_q'] = stats.gamma.ppf(dm_inv_fdc_t['cr'],a=dm_inv_fdc_t['para1']*2,scale=dm_inv_fdc_t['para2'])
                    dm_inv_fdc_t['nbinom_q'] = stats.nbinom.ppf(dm_inv_fdc_t['cr'],dm_inv_fdc_t['para1']*2,dm_inv_fdc_t['para2'])
                    dm_inv_fdc_t['nbinom_q'].fillna(0,inplace=True)
                    dm_inv_fdc_t['demand_q'] = dm_inv_fdc_t['gamma_q'] * dm_inv_fdc_t['flag_gm'] + dm_inv_fdc_t['nbinom_q'] * dm_inv_fdc_t['flag_nb']

                    dm_inv_fdc_t['transfer_x'] = dm_inv_fdc_t['demand_q'] - dm_inv_fdc_t['stock_quantity']
                    dm_inv_fdc_t['transfer_x'] = dm_inv_fdc_t.transfer_x.apply(lambda x: x if x>0 else 0)
                    
                    # 再和最大目标库存对应的最大调拨量取Min
                    dm_inv_fdc_t['transfer_x'] = dm_inv_fdc_t[['transfer_x','max_transfer']].min(axis=1)
                    
                    # 四舍五入取整，再和可分配量取min，得到真正的调拨决策量
                    dm_inv_fdc_t['transfer_x'] = np.round(dm_inv_fdc_t['transfer_x'])
                    dm_inv_fdc_t['transfer_x'] = dm_inv_fdc_t[['transfer_x','capacity_interger']].min(axis=1)

                    sku_cnt_fdc_t = dm_inv_fdc_t[dm_inv_fdc_t['transfer_x']>0]['transfer_x'].count()
                    capacity_cnt_fdc_t = dm_inv_fdc_t[dm_inv_fdc_t['transfer_x']>0]['transfer_x'].sum()

                    if (sku_cnt_fdc_t<=sku_lm) and (capacity_cnt_fdc_t<=cp_lm):
                        # 没有超过时，记录在dm_invntry_fdc_i中，之后超过会回退一步
                        dm_inv_fdc_i = dm_inv_fdc_t.copy()
                        #m_cost_sku_cr += step_cr
                        
                    if (sku_cnt_fdc_t>sku_lm) or (capacity_cnt_fdc_t>cp_lm):
                        # 看是哪个限制达到了。然后回退一步，数据和cr都回退
                        sku_flag = (sku_cnt_fdc_t > sku_lm)
                        capacity_flag = (capacity_cnt_fdc_t>cp_lm)
                        dm_inv_fdc_t = dm_inv_fdc_i.copy()
                        m_cost_sku_cr -= step_cr
                        break

            if (capacity_flag == True):
                # 有可能是两个限制都超了啊？这里不管，但从打印的结果看，基本上都是SKU超过了。
                dm_inv_fdc_i['allocation'] = dm_inv_fdc_i['transfer_x']
                dm_invntry_fdc_cp = dm_inv_fdc_i.copy()
            else:
                k = 0
                dm_inv_fdc_i['flag1'] = dm_inv_fdc_i.transfer_x.apply(lambda x: 1 if x>0 else 0)
                # 这里改了，最大的cost不用重新选，因为想要继承上面的m_cost_sku_cr，不过影响不大
                #dm_inv_fdc_i['m_cost_cp'] = dm_inv_fdc_i[dm_inv_fdc_i['flag1']==1].cost_adj.max()
                dm_inv_fdc_i['m_cost_cp'] = dm_inv_fdc_i.cost_adj.max()
                dm_invntry_fdc_cp = dm_inv_fdc_i.copy()
                #dm_inv_fdc_i['m_cost_cp'] = dm_inv_fdc_i[dm_inv_fdc_i['flag1']==1].stockout_cost.max()
                dm_invntry_fdc_cp['transfer_x_cp'] = dm_invntry_fdc_cp['transfer_x']
                # 从上一步的cr开始，减少这一步的循环次数。
                m_cost_sku_cr_cp = m_cost_sku_cr - step_cr
                step_cr_cp = (1-m_cost_sku_cr_cp)/2
                # print('initial m_cost_sku_cr',m_cost_sku_cr)
                while step_cr_cp > 0.0001:  # 这里要比前面那个小！很重要
                    
                    step_cr_cp = step_cr_cp/2
                    while True:
                        #print('FDC',i,'m_cost_sku_cr',m_cost_sku_cr,'step_cr',step_cr,'sku_cnt_fdc_t',sku_cnt_fdc_t,'sku_lm',sku_lm,'capacity_cnt_fdc_t',capacity_cnt_fdc_t,'cp_lm',cp_lm)
                        k += 1
                        m_cost_sku_cr_cp += step_cr_cp
                        dm_inv_fdc_i['m_cost_sku_cr_cp'] = m_cost_sku_cr_cp
                        dm_inv_fdc_i['cr_cp'] = 1-dm_inv_fdc_i['m_cost_cp']/dm_inv_fdc_i['cost_adj']*(1-dm_inv_fdc_i['m_cost_sku_cr_cp'])
                        #dm_inv_fdc_i['cr'] = 1-dm_inv_fdc_i['m_cost_cp']/dm_inv_fdc_i['stockout_cost']*(1-dm_inv_fdc_i['m_cost_sku_cr_cp'])

                        dm_inv_fdc_i['cr_cp'] = dm_inv_fdc_i['cr_cp'].apply(lambda x: 1 if x>1 else (0 if x<0 else x))

                        dm_inv_fdc_i['gamma_q_cp'] = stats.gamma.ppf(dm_inv_fdc_i['cr_cp'],a=dm_inv_fdc_i['para1']*2,scale=dm_inv_fdc_i['para2'])
                        dm_inv_fdc_i['nbinom_q_cp'] = stats.nbinom.ppf(dm_inv_fdc_i['cr_cp'],dm_inv_fdc_i['para1']*2,dm_inv_fdc_i['para2'])
                        dm_inv_fdc_i['nbinom_q_cp'].fillna(0,inplace=True)
                        dm_inv_fdc_i['demand_q_cp'] = dm_inv_fdc_i['gamma_q_cp'] * dm_inv_fdc_i['flag_gm'] + dm_inv_fdc_i['nbinom_q_cp'] * dm_inv_fdc_i['flag_nb']

                        dm_inv_fdc_i['transfer_x_cp'] = dm_inv_fdc_i['demand_q_cp'] - dm_inv_fdc_i['stock_quantity']
                        dm_inv_fdc_i['transfer_x_cp'] = dm_inv_fdc_i.transfer_x_cp.apply(lambda x: x if x>0 else 0)
                        dm_inv_fdc_i['transfer_x_cp'] = dm_inv_fdc_i['transfer_x_cp'] * dm_inv_fdc_i['flag1']
                        
                        # 再和最大目标库存对应的最大调拨量取Min
                        dm_inv_fdc_i['transfer_x_cp'] = dm_inv_fdc_i[['transfer_x_cp','max_transfer']].min(axis=1)
                        
                        # 四舍五入取整
                        dm_inv_fdc_i['transfer_x_cp'] = np.round(dm_inv_fdc_i['transfer_x_cp'])
                        dm_inv_fdc_i['transfer_x_cp'] = dm_inv_fdc_i[['transfer_x_cp','capacity_interger']].min(axis=1)
                        capacity_cnt_fdc_i = dm_inv_fdc_i[dm_inv_fdc_i['transfer_x_cp']>0]['transfer_x_cp'].sum()
                        
                        if (capacity_cnt_fdc_i<=cp_lm):
                            dm_invntry_fdc_cp = dm_inv_fdc_i.copy()
                            #m_cost_sku_cr_cp += step_cr_cp

                        if (capacity_cnt_fdc_i>cp_lm):
                            # 此处i才是变动的了
                            dm_inv_fdc_i = dm_invntry_fdc_cp.copy()
                            m_cost_sku_cr_cp -= step_cr_cp
                            break
                dm_invntry_fdc_cp['allocation'] = dm_invntry_fdc_cp[['transfer_x_cp','capacity_interger']].min(axis=1)
                dm_invntry_fdc_cp['allocation'].fillna(0, inplace=True)
                
            # print('dc:',i ,'capacity_flag:',capacity_flag)
            allocation_fdc_cp = dm_invntry_fdc_cp.loc[:,['item_sku_id','dc_id','allocation']].copy()
            return allocation_fdc_cp


        def allocation_fdc(init_cr,limit,inventory,allocation_rdc_r):

            timestart = time.time()
            partial_param = functools.partial(allocation_fdc_i,init_cr,limit,inventory,allocation_rdc_r)
            pool = mp.Pool(processes=5)   # 4进程55s，5线程48s，
            #pool = Pool(processes=5)
            try:
                fdc_output = pool.map(partial_param, range(1,6))
                pool.close()
                #pool.join()
            except KeyboardInterrupt as e:
                pool.terminate()
                raise e
            temp = fdc_output[0]
            for j in range(1,5):
                temp = temp.append(fdc_output[j])
            print("single dc multi sku problem:",time.time() - timestart)
            return temp

        allocation_rdc_r = allocation_rdc(inventory,replenish,t)

        allocation_fdc_r = allocation_fdc(0.01,limit,inventory,allocation_rdc_r)
        distribution = pd.merge(allocation_fdc_r, allocation_rdc_r, how='inner', on=('item_sku_id','dc_id'))
        distribution['daily_decision'] = distribution[['allocation','capacity_interger']].min(axis=1)

        distribution_r = distribution.sort_values(['dc_id', 'item_sku_id'], ascending=[True, True])

        transshipment_decision = np.zeros((1000, 5))
        for i in range(0,5):
            transshipment_decision[:,i] = distribution_r.loc[distribution_r['dc_id']==i+1]['daily_decision'].values
        transshipment_decision = np.transpose(transshipment_decision)

        return transshipment_decision.astype(int)


    def info_update(self,end_day_inventory,t):
        '''
        input values: inventory information at the end of day t
        '''
        self.inv.append(end_day_inventory)

    def some_other_functions():
        pass